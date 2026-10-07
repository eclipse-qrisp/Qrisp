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

"""Defines QuantumModulus, a QuantumFloat subtype for modular arithmetic used in e.g. Shor's algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeGuard, TypeVar, cast

import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.core import Tracer

from qrisp import check_for_tracing_mode
from qrisp.core import QuantumSession, cx, measure
from qrisp.environments import invert
from qrisp.jasp import q_fori_loop
from qrisp.misc import gate_wrap
from qrisp.qtypes.quantum_bool import QuantumBool
from qrisp.qtypes.quantum_float import QuantumFloat

if TYPE_CHECKING:
    # Type checking only: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    # QuantumModulus supports integer operands only (traced integers are jax
    # Arrays), so its operators narrow the QuantumFloat signatures.
    _ClassicalInteger = int | np.integer | Array | BigInteger
    _Operand = QuantumFloat | _ClassicalInteger
    _ComparisonOperand = QuantumFloat | int | np.integer | Array

_Comparison = TypeVar("_Comparison", bound=Callable[..., "QuantumBool"])


def _moduli_neq(a, b):
    """Compare two moduli for inequality.

    This check is designed for **static** (non-tracing) contexts only.
    When either operand contains JAX tracers (dynamic mode), the digits
    cannot be materialised, so the function raises ``RuntimeError``.
    Callers in tracing code paths must skip the
    check themselves (``if not check_for_tracing_mode(): …``).

    Parameters
    ----------
    a : int or BigInteger
        First modulus to compare.
    b : int or BigInteger
        Second modulus to compare.

    Returns
    -------
    bool
        ``True`` if the moduli are unequal, ``False`` if they represent
        the same modulus.

    """
    # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    if isinstance(a, BigInteger) or isinstance(b, BigInteger):
        if not isinstance(a, BigInteger):
            a = BigInteger.create_static(a, b.digits.shape[0])
        elif not isinstance(b, BigInteger):
            b = BigInteger.create_static(b, a.digits.shape[0])

        if isinstance(a.digits, Tracer) or isinstance(b.digits, Tracer):
            raise RuntimeError(
                "_moduli_neq cannot compare traced moduli."
                "This check should only be called in static (non-tracing) "
                "mode.  Guard the call with `if not check_for_tracing_mode():`."
            )

        a_digits = np.asarray(a.digits)
        b_digits = np.asarray(b.digits)
        if a_digits.shape != b_digits.shape:
            return True
        return bool(np.any(a_digits != b_digits))

    return a != b


def _coerce_bigint_operand(value, modulus):
    """Coerce *value* to a BigInteger matching *modulus*'s limb count.

    If *value* is already a BigInteger with fewer limbs it is zero-padded;
    if it has more limbs than the modulus a ``ValueError`` is raised.
    Non-BigInteger values are converted via ``BigInteger.coerce``.

    Parameters
    ----------
    value : int, BigInteger, or array-like
        Operand to coerce.
    modulus : BigInteger
        Reference modulus whose limb count determines the target width.

    Returns
    -------
    BigInteger
        Value with exactly ``modulus.digits.shape[0]`` limbs.

    """
    # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    return BigInteger.coerce(value, modulus.digits.shape[0])


def _is_classical_factor(value: object) -> TypeGuard[_ClassicalInteger]:
    """Return whether ``value`` is a classical integer that a QuantumModulus can be multiplied with."""
    # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

    return isinstance(value, (int, np.integer, jnp.integer, Array, Tracer, BigInteger))


def _normalize_modular_arithmetic_operand(qm: QuantumModulus, other: _Operand) -> _Operand:
    """Return a quantum operand as is, and encode a classical one in the representation of ``qm``."""
    if isinstance(other, QuantumFloat):
        return other

    return qm.encoder(other % qm.modulus)


def comparison_wrapper(func: _Comparison) -> _Comparison:
    """Check that a QuantumModulus comparison compares values in the same representation, then run it."""

    @wraps(func)
    def res_func(self: QuantumModulus, other: Any) -> QuantumBool:

        if not check_for_tracing_mode() and isinstance(other, QuantumModulus):
            # Two QuantumModuli can be compared as long as they share the
            # same Montgomery shift (they're in the same representation).
            if self.m != other.m:
                raise ValueError("Tried to evaluate QuantumModulus comparison with differing Montgomery shifts")

            if _moduli_neq(self.modulus, other.modulus):
                raise ValueError("Tried to compare QuantumModulus instances of differing modulus")
        elif not check_for_tracing_mode() and self.m != 0:
            # Comparing against a non-QuantumModulus (e.g. QuantumFloat)
            # requires standard representation (m == 0).
            raise ValueError("Tried to evaluate QuantumModulus comparison with non-zero Montgomery shift")

        return func(self, other)

    return cast(_Comparison, res_func)


class QuantumModulus(QuantumFloat):
    r"""A subtype of :ref:`QuantumFloat` for modular arithmetic, as used in Shor's algorithm.

    It can be used to model and process
    `modular arithmetic <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.
    Modular arithmetic plays an important role in many cryptographical applications,
    especially in Shor's algorithm.

    The QuantumModulus allows users convenient access to modular arithmetic, which
    for many gate based frameworks is rather complicated due to the intricacies
    of the reduction process.

    For a first example we simply add two instances:

    >>> from qrisp import QuantumModulus
    >>> N = 13
    >>> a = QuantumModulus(N)
    >>> b = QuantumModulus(N)
    >>> a[:] = 5
    >>> b[:] = 10
    >>> c = a + b
    >>> print(c)
    {2: 1.0}

    We get the output 2 because

    >>> (5 + 10)%13
    2

    Similar to :ref:`QuantumFloat`, subtraction and addition are also supported:

    >>> d = a*b
    >>> print(d)
    {11: 1.0}

    Check the result:

    >>> (5*10)%13
    11

    Especially relevant for Shor's algorithm are the in-place operations:

    >>> a = QuantumModulus(N)
    >>> a[:] = 5
    >>> a *= 10
    >>> print(a)
    {11: 1.0}

    **Specifying a custom adder**

    It is possible to specify a custom adder that is used when processing the
    Modular-Arithmetic. For this, you can use the ``inpl_adder`` keyword.

    By default, the `Fourier-adder <https://arxiv.org/abs/quant-ph/0008033>`_
    is used, but we can for instance also try the `Cuccaro-adder <https://arxiv.org/abs/quant-ph/0410184>`_.

    >>> from qrisp import cuccaro_adder
    >>> a = QuantumModulus(N, inpl_adder = cuccaro_adder)
    >>> a[:] = 5
    >>> a *= 10
    >>> print(a)
    {11: 1.0}

    Or the `Gidney-adder <https://arxiv.org/abs/1709.06648>`_.

    >>> from qrisp import gidney_adder
    >>> a = QuantumModulus(N, inpl_adder = gidney_adder)
    >>> a[:] = 5
    >>> a *= 10
    >>> print(a)
    {11: 1.0}

    To learn how to create your own adder for this feature, please visit :meth:`this page <qrisp.inpl_adder_test>`.

    .. warning::

        Currently the adder is only used in in-place multiplication, since this is the
        relevant operation for Shor's algorithm. The other operations (such as addition etc.)
        will follow in a future release of Qrisp.

    **Advanced usage**

    The modular multiplication uses a technique called `Montgomery reduction <https://en.wikipedia.org/wiki/Montgomery_modular_multiplication>`_.
    The quantum version of this algorithm can be found in `this paper <https://arxiv.org/abs/1801.01081>`__.
    The idea behind Montgomery reduction is to choose a differing representation of numbers to enhance
    the reduction step of modular arithmetic. This representation works as follows:
    For an integer $m$ called Montgomery shift, the modular number $a \in \mathbb{Z}/N\mathbb{Z}$ is represented as

    .. math::

        \hat{k} = (2^{-m} k) \text{mod} N

    If you're interested in why this representation is advantageous, we recommend
    checking out the linked resources above.

    For Qrisp, the Montgomery shift can be modified via the attribute ``m``.

    >>> a = QuantumModulus(N)
    >>> a.m = 3
    >>> a[:] = 1
    >>> print(a)
    {1: 1.0}

    We shift back to 0:

    >>> a.m -= 3
    >>> print(a)
    {8: 1.0}

    Note that this shift is only a compiler shift - ie. no quantum gates are applied.
    Instead the :meth:`decoder <qrisp.QuantumVariable.decoder>` function is modified.

    """

    def __init__(
        self,
        modulus: int | BigInteger | Array,
        inpl_adder: Callable | None = None,
        qs: QuantumSession | None = None,
    ) -> None:
        """Create a register holding values modulo ``modulus``, multiplied with the adder ``inpl_adder``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import smallest_power_of_two

        self.modulus = modulus
        aux = smallest_power_of_two(modulus)
        QuantumFloat.__init__(self, msize=aux, qs=qs)
        # A traced modulus must be threaded through the pytree as a proper leaf,
        # otherwise it gets smuggled into static aux_data (via QuantumVariableTemplate's
        # copy.copy(qv)) and leaks stale tracers across independent traces.
        self.traced_attributes = self.traced_attributes + ["modulus"]
        # Dynamic modular arithmetic uses gidney_adder while static uses fourier_adder.
        # Consider reconciling it.
        if check_for_tracing_mode():
            if inpl_adder is None:
                # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
                from qrisp.alg_primitives.arithmetic.adders import gidney_adder

                inpl_adder = gidney_adder

        elif inpl_adder is None:
            # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
            from qrisp.alg_primitives.arithmetic import fourier_adder

            inpl_adder = fourier_adder

        self.inpl_adder = inpl_adder
        self.m = 0

    def decoder(self, i):
        """Decode a Montgomery-encoded value back to standard representation.

        Montgomery encoding stores k as k_hat = (k * 2^{-m}) mod N.
        Decoding recovers k = (k_hat * 2^m) mod N = montgomery_decoder(k_hat, R, N)
        where R = 2^m is the Montgomery radix.
        """
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import new_montgomery_decoder

        if check_for_tracing_mode():
            return new_montgomery_decoder(i, self.m, self.modulus)

        i_value = i() if isinstance(i, BigInteger) else int(i)
        modulus_value = self.modulus() if isinstance(self.modulus, BigInteger) else int(self.modulus)

        if i_value >= modulus_value:
            return np.nan

        return new_montgomery_decoder(i, self.m, self.modulus)

    def jdecoder(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, i: int | BigInteger | Array
    ) -> int | BigInteger | Array:
        """Decode a Montgomery-encoded value, also under tracing and without the range check of ``decoder``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import new_montgomery_decoder

        return new_montgomery_decoder(i, self.m, self.modulus)

    def measure(self) -> int | BigInteger | Array:
        """Measure the register and return the decoded value, as a BigInteger for a BigInteger modulus."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

        if not isinstance(self.modulus, BigInteger):
            return self.jdecoder(self.reg.measure())

        if check_for_tracing_mode():
            for_loop = q_fori_loop
        else:

            def for_loop(lower: int, upper: int, body_fun: Callable[[int, Array], Array], init_val: Array) -> Array:
                val = init_val
                for i in range(lower, upper):
                    val = body_fun(i, val)
                return val

        def body_fun(i: int | Array, val: Array) -> Array:
            return val.at[i].set(measure(self[32 * i : 32 * (i + 1)]).astype(jnp.uint32))

        digits = for_loop(0, (self.size - 1) // 32, body_fun, jnp.zeros_like(self.modulus.digits))
        digits = digits.at[(self.size - 1) // 32].set(measure(self[32 * ((self.size - 1) // 32) :]).astype(jnp.uint32))
        return self.jdecoder(BigInteger(digits))

    def encoder(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, i: _ClassicalInteger
    ) -> int | BigInteger | Array:
        """Encode a value in the Montgomery representation of the register.

        Outside of tracing, the value must lie in ``[0, modulus)``.
        """
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

        if check_for_tracing_mode():
            # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import montgomery_encoder

            if isinstance(i, BigInteger):
                return montgomery_encoder(i, BigInteger.create(1, i.digits.shape[0]) << self.m, self.modulus)
            return montgomery_encoder(i, 1 << self.m, self.modulus)

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import montgomery_encoder

        i_value = i() if isinstance(i, BigInteger) else int(i)
        modulus_value = self.modulus() if isinstance(self.modulus, BigInteger) else int(self.modulus)

        if i_value >= modulus_value:
            raise ValueError("Tried to encode a number into QuantumModulus, which is greater or equal to the modulus")
        if i_value < 0:
            raise ValueError("Tried to encode a negative number into QuantumModulus")

        return montgomery_encoder(i_value, 2**self.m, modulus_value)

    def _classical_operand(self, value: _ClassicalInteger) -> int | BigInteger | Array:
        """Return a classical factor in the form that the Montgomery multiplication expects.

        For a BigInteger modulus, the factor is returned with twice the limbs of the
        modulus, so that the intermediate products cannot overflow.

        Parameters
        ----------
        value : int, Array or BigInteger
            The classical factor.

        Returns
        -------
        int, Array or BigInteger
            The factor, as a Python int for a NumPy integer.

        """
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

        # Convert np.integer to a Python int for the Montgomery helpers
        if isinstance(value, np.integer):
            value = int(value)

        if isinstance(self.modulus, BigInteger):
            if not isinstance(value, BigInteger):
                value = _coerce_bigint_operand(value, self.modulus)
            return value.get_larger()
        return value

    def _classical_factor_operands(
        self, other: _ClassicalInteger
    ) -> tuple[int | BigInteger | Array, int | BigInteger | Array, int | Array]:
        """Return the factor, the modulus and the Montgomery shift of the multiplication by a classical factor.

        For a BigInteger modulus, the factor and the modulus are returned with
        twice the limbs, so that the intermediate products cannot overflow.
        """
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import best_montgomery_shift

        # The Montgomery reduction sums one partial product (reduced modulo N)
        # per qubit of self, so the shift depends on the size of self, not on
        # the value of the classical factor.
        shift = best_montgomery_shift(self.size, self.modulus)

        modulus = self.modulus.get_larger() if isinstance(self.modulus, BigInteger) else self.modulus
        return self._classical_operand(other), modulus, shift

    def _multiply_inplace(self, factor: _ClassicalInteger, inverse: _ClassicalInteger | None) -> None:
        """Multiply the register in place by a classical factor, with its inverse modulo N if it is given.

        Parameters
        ----------
        factor : int, Array or BigInteger
            The classical factor.
        inverse : int, Array or BigInteger, optional
            The inverse of the factor modulo N, or None to compute it.

        """
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply_inplace

        factor_operand, modulus, shift = self._classical_factor_operands(factor)
        inverse_operand = None if inverse is None else self._classical_operand(inverse)
        cq_montgomery_multiply_inplace(factor_operand, self, modulus, shift, self.inpl_adder, X_inverse=inverse_operand)

    @gate_wrap(permeability="args", is_qfree=True)
    def __mul__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: QuantumModulus | _ClassicalInteger
    ) -> QuantumFloat:
        """Return a new QuantumModulus holding the product with a QuantumModulus or a classical factor."""
        if isinstance(other, QuantumModulus):
            if not check_for_tracing_mode() and _moduli_neq(self.modulus, other.modulus):
                raise ValueError("Both QuantumModuli must have the same modulus")
            if check_for_tracing_mode():
                # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
                from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
                    qq_montgomery_multiply_modulus,
                )

                return qq_montgomery_multiply_modulus(self, other)

            # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
            from qrisp.alg_primitives.arithmetic.modular_arithmetic import montgomery_mod_mul

            return montgomery_mod_mul(self, other)

        if _is_classical_factor(other):
            # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import cq_montgomery_multiply

            factor, modulus, shift = self._classical_factor_operands(other)
            return cq_montgomery_multiply(factor, self, modulus, shift)

        raise TypeError(f"Quantum modular multiplication with type {type(other)} not implemented")

    __rmul__ = __mul__

    @gate_wrap(permeability=[1], is_qfree=True)
    def __imul__(self, other: _ClassicalInteger) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Multiply the register in place by a classical factor."""
        if _is_classical_factor(other):
            self._multiply_inplace(other, None)
            return self

        raise TypeError(f"Quantum modular in-place multiplication with type {type(other)} not implemented")

    @gate_wrap(permeability=[1], is_qfree=True)
    def inpl_mult(self, factor: _ClassicalInteger, inverse: _ClassicalInteger | None = None) -> QuantumModulus:
        r"""Multiply the register in place by a classical factor, optionally with its precomputed inverse.

        ``qm.inpl_mult(X)`` is the same as ``qm *= X``. To uncompute a temporary
        register, the in-place multiplication also multiplies by the inverse of
        $X$ modulo $N$, and computing this inverse is its most expensive classical
        step. If the inverse is known in advance, for example for the powers
        $a^{2^i}$ of Shor's algorithm, passing it skips this computation. The
        circuit is the same.

        Parameters
        ----------
        factor : int, Array or BigInteger
            The classical factor $X$, coprime to the modulus. NumPy integers are
            accepted as well.
        inverse : int, Array or BigInteger, optional
            The inverse of $X$ modulo $N$. It is not checked. The default is None,
            which computes it.

        Returns
        -------
        QuantumModulus
            The register, multiplied in place.

        Raises
        ------
        TypeError
            If the factor or the inverse is not a classical integer.

        Examples
        --------
        Multiply by 7 modulo 15, whose inverse modulo 15 is 13:

        >>> from qrisp import QuantumModulus
        >>> qm = QuantumModulus(15)
        >>> qm[:] = 2
        >>> qm.inpl_mult(7, inverse=13)  # doctest: +SKIP
        >>> print(qm)
        {14: 1.0}

        """
        if not _is_classical_factor(factor) or not (inverse is None or _is_classical_factor(inverse)):
            raise TypeError(
                f"Quantum modular in-place multiplication with types {type(factor)}, {type(inverse)} not implemented"
            )
        self._multiply_inplace(factor, inverse)
        return self

    @gate_wrap(permeability="args", is_qfree=True)
    def __add__(self, other: _Operand) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a new QuantumModulus holding the sum with a quantum or classical value."""
        other = _normalize_modular_arithmetic_operand(self, other)
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise ValueError("Tried to add two QuantumModulus with differing Montgomery shift")
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise ValueError("Tried to add a QuantumFloat and QuantumModulus with non-zero Montgomery shift")

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        res = self.duplicate()
        cx(self, res)

        mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    __radd__ = __add__

    @gate_wrap(permeability=[1], is_qfree=True)
    def __iadd__(self, other: _Operand) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Add a quantum or classical value to the register in place."""
        other = _normalize_modular_arithmetic_operand(self, other)
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
                from qrisp.alg_primitives.arithmetic.modular_arithmetic import montgomery_addition

                montgomery_addition(other, self)
                return self
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise ValueError("Tried to add a QuantumFloat and QuantumModulus with non-zero Montgomery shift")

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        mod_adder(other, self, self.inpl_adder, self.modulus)
        return self

    @gate_wrap(permeability="args", is_qfree=True)
    def __sub__(self, other: _Operand) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a new QuantumModulus holding the difference with a quantum or classical value."""
        other = _normalize_modular_arithmetic_operand(self, other)
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise ValueError("Tried to subtract QuantumModulus with differing Montgomery shift")
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise ValueError("Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift")

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        res = self.duplicate()
        cx(self, res)

        with invert():
            mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    @gate_wrap(permeability="args", is_qfree=True)
    def __rsub__(self, other: _Operand) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a new QuantumModulus holding a quantum or classical value minus the register."""
        other = _normalize_modular_arithmetic_operand(self, other)
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise ValueError("Tried to subtract QuantumModulus with differing Montgomery shift")
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise ValueError("Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift")

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        res = self.duplicate()

        res -= self

        mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    @gate_wrap(permeability=[1], is_qfree=True)
    def __isub__(self, other: _Operand) -> QuantumModulus:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Subtract a quantum or classical value from the register in place."""
        other = _normalize_modular_arithmetic_operand(self, other)
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise ValueError("Tried to subtract QuantumModulus with differing Montgomery shift")
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise ValueError("Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift")

        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        with invert():
            mod_adder(other, self, self.inpl_adder, self.modulus)

        return self

    @comparison_wrapper
    def __lt__(self, other: _ComparisonOperand) -> QuantumBool:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a QuantumBool indicating whether the register is less than ``other``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives import uint_lt

        return uint_lt(self, other, self.inpl_adder)

    @comparison_wrapper
    def __gt__(self, other: _ComparisonOperand) -> QuantumBool:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a QuantumBool indicating whether the register is greater than ``other``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives import uint_gt

        return uint_gt(self, other, self.inpl_adder)

    @comparison_wrapper
    def __le__(self, other: _ComparisonOperand) -> QuantumBool:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a QuantumBool indicating whether the register is at most ``other``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives import uint_le

        return uint_le(self, other, self.inpl_adder)

    @comparison_wrapper
    def __ge__(self, other: _ComparisonOperand) -> QuantumBool:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Return a QuantumBool indicating whether the register is at least ``other``."""
        # Imported here: qrisp.alg_primitives imports qrisp, which imports this module (circular import)
        from qrisp.alg_primitives import uint_ge

        return uint_ge(self, other, self.inpl_adder)

    @comparison_wrapper
    def __eq__(self, other: object) -> QuantumBool:
        """Return a QuantumBool indicating whether the register equals ``other``."""
        return QuantumFloat.__eq__(self, other)

    @comparison_wrapper
    def __ne__(self, other: object) -> QuantumBool:
        """Return a QuantumBool indicating whether the register differs from ``other``."""
        return QuantumFloat.__ne__(self, other)

    def __hash__(self) -> int:
        """Hash by identity, since the equality operator returns a QuantumBool."""
        return id(self)
