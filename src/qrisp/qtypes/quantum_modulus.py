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
import jax.numpy as jnp

from qrisp import check_for_tracing_mode
from qrisp.qtypes.quantum_float import QuantumFloat
from qrisp.misc import gate_wrap
from qrisp.core import QuantumVariable
import jax


def comparison_wrapper(func):

    def res_func(self, other):

        if self.m != 0:
            raise Exception(
                "Tried to evaluate QuantumModulus comparison with non-zero Montgomery shift"
            )

        conversion_flag = False
        if isinstance(other, QuantumModulus):

            if other.m != 0:
                raise Exception(
                    "Tried to evaluate QuantumModulus comparison with non-zero Montgomery shift"
                )

            if self.modulus != other.modulus:
                raise Exception(
                    "Tried to compare QuantumModulus instances of differing modulus"
                )

            other.__class__ = QuantumFloat
        self.__class__ = QuantumFloat
        res = func(self, other)
        self.__class__ = QuantumModulus
        if conversion_flag:
            other.__class__ = QuantumModulus
        return res

    return res_func


class QuantumModulus(QuantumFloat):
    r"""
    This class is a subtype of :ref:`QuantumFloat`, which can be used to model and
    process `modular arithmetic <https://en.wikipedia.org/wiki/Modular_arithmetic>`_.
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

    def __init__(self, modulus, inpl_adder=None, qs=None):

        if check_for_tracing_mode():
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import smallest_power_of_two
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
            if isinstance(modulus, BigInteger):
                pad = jnp.zeros(modulus.digits.shape[0], dtype=modulus.digits.dtype)
                modulus = BigInteger(jnp.concatenate([modulus.digits, pad], axis=0))
            self.modulus = modulus
            aux = smallest_power_of_two(modulus)

            QuantumFloat.__init__(self, msize=aux, qs=qs)
            if inpl_adder is None:
                from qrisp.alg_primitives.arithmetic import gidney_adder
                inpl_adder = gidney_adder

            self.inpl_adder = inpl_adder

            self.m = 0

        else:
            self.modulus = modulus
            self.m = int(np.ceil(np.log2(modulus)))

            QuantumFloat.__init__(self, msize=self.m, qs=qs)

            if inpl_adder is None:
                from qrisp.alg_primitives.arithmetic import fourier_adder

                inpl_adder = fourier_adder

            self.inpl_adder = inpl_adder

            self.m = 0

    def decoder(self, i):
        if check_for_tracing_mode():
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
                BigInteger
            )
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
                montgomery_decoder
            )

            if isinstance(i, BigInteger):
                n = i.digits.shape[0]
                R = BigInteger.create(1, n) << self.m
                return montgomery_decoder(i, R, self.modulus)
            
            else:
                from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import montgomery_decoder
                return montgomery_decoder(i, 2**self.m, self.modulus)
        

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import (
            montgomery_decoder,
        )

        if i >= self.modulus:  # or (np.gcd(i, self.modulus) != 1 and i != 0):
            return np.nan
        return montgomery_decoder(i, 2 ** self.m, self.modulus)
    
    def jdecoder(self, i):
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import montgomery_decoder
        return montgomery_decoder(i, 2 ** self.m, self.modulus)
    
    def measure(self):
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
            BigInteger
        )
        if isinstance(self.modulus, BigInteger):
            from qrisp import q_fori_loop, measure, jlen
            if check_for_tracing_mode():
                for_loop = q_fori_loop
            else:
                def for_loop(lower, upper, body_fun, init_val):
                    val = init_val
                    for i in range(lower, upper):
                        val = body_fun(i, val)
                    return val
            def body_fun(i, val):
                return val.at[i].set(measure(self[32*i:32*(i+1)]))
            digits = for_loop(0, (self.size-1)//32, body_fun, jnp.zeros_like(self.modulus.digits))
            digits = digits.at[(self.size-1)//32].set(measure(self[32*((self.size - 1)//32):]))
            return self.jdecoder(BigInteger(digits))
        else:
            return self.jdecoder(self.reg.measure())

    def encoder(self, i):
        if check_for_tracing_mode():
        
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import (
                BigInteger
            )
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import montgomery_encoder
            if isinstance(i, BigInteger):
                return montgomery_encoder(i, BigInteger.create(1, i.digits.shape[0]) << self.m, self.modulus)
            else:
                return montgomery_encoder(i, 1 << self.m, self.modulus)
        
        else:

            if i >= self.modulus:
                raise Exception(
                    "Tried to encode a number into QuantumModulus, which is greator or equal to the modulus"
                )
            if i < 0:
                raise Exception("Tried to encode a negative number into QuantumModulus")

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import (
            montgomery_encoder,
        )

        # if i >= self.modulus:  # or (np.gcd(i, self.modulus) != 1 and i != 0):
        #     return np.nan

        return montgomery_encoder(i, 2**self.m, self.modulus)

    #def encode(self, i):
    #    QuantumVariable.encode(self, self.encoder(i))

    @gate_wrap(permeability="args", is_qfree=True)
    def __mul__(self, other):
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
        if isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise ValueError("Both QuantumModuli must have the same shift")
            if self.modulus != other.modulus:
                raise ValueError("Both QuantumModuli must have the same modulus")
            if check_for_tracing_mode():
                from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
                    qq_montgomery_multiply_modulus
                )
                return qq_montgomery_multiply_modulus(self, other)
            else:
                from qrisp.alg_primitives.arithmetic.modular_arithmetic import (
                    montgomery_mod_mul,
                )           
                return montgomery_mod_mul(self, other)

        elif isinstance(other, (int, np.integer, jnp.integer, BigInteger)):
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
                cq_montgomery_multiply
            )
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
                best_montgomery_shift
            )
            if isinstance(other, BigInteger):
                pad = jnp.zeros(other.digits.shape[0], dtype=other.digits.dtype)
                other = BigInteger(jnp.concatenate([other.digits, pad], axis=0))
            shift = best_montgomery_shift(other, self.modulus)
            return cq_montgomery_multiply(other, self, self.modulus, shift)
        else:
            raise Exception(
                f"Quantum modular multiplication with type {type(other)} not implemented"
            )

    __rmul__ = __mul__

    @gate_wrap(permeability=[1], is_qfree=True)
    def __imul__(self, other):
        from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger

        if isinstance(other, (int, np.integer, jnp.integer, jax.Array, BigInteger)):
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_montgomery import (
                cq_montgomery_multiply_inplace
            )
            from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
                best_montgomery_shift
            )
            if isinstance(other, BigInteger):
                pad = jnp.zeros(other.digits.shape[0], dtype=other.digits.dtype)
                other = BigInteger(jnp.concatenate([other.digits, pad], axis=0))
            
            shift = best_montgomery_shift(other, self.modulus)
            cq_montgomery_multiply_inplace(other, self, self.modulus, shift, self.inpl_adder)
            return self
        else: 
            raise Exception(
                f"Quantum modular in-place multiplication with type {type(other)} not implemented"
            )


    @gate_wrap(permeability="args", is_qfree=True)
    def __add__(self, other):
        if isinstance(other, int):
            other = self.encoder(other % self.modulus)
        elif isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise Exception(
                    "Tried to add two QuantumModulus with differing Montgomery shift"
                )
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise Exception(
                    "Tried to add a QuantumFloat and QuantumModulus with non-zero Montgomery shift"
                )

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        res = self.duplicate(init=True)

        mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    __radd__ = __mul__

    @gate_wrap(permeability=[1], is_qfree=True)
    def __iadd__(self, other):
        if isinstance(other, int):
            other = self.encoder(other % self.modulus)
        elif isinstance(other, QuantumModulus):
            if self.m != other.m:
                from qrisp.alg_primitives.arithmetic.modular_arithmetic import (
                    montgomery_addition,
                )

                montgomery_addition(other, self)
                return self
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise Exception(
                    "Tried to add a QuantumFloat and QuantumModulus with non-zero Montgomery shift"
                )

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        mod_adder(other, self, self.inpl_adder, self.modulus)
        return self

    @gate_wrap(permeability="args", is_qfree=True)
    def __sub__(self, other):
        if isinstance(other, int):
            other = self.encoder(other % self.modulus)
        elif isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise Exception(
                    "Tried to add subtract QuantumModulus with differing Montgomery shift"
                )
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise Exception(
                    "Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift"
                )

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder
        from qrisp.environments import invert

        res = self.duplicate(init=True)

        with invert():
            mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    @gate_wrap(permeability="args", is_qfree=True)
    def __rsub__(self, other):
        if isinstance(other, int):
            other = self.encoder(other % self.modulus)
        elif isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise Exception(
                    "Tried to add subtract QuantumModulus with differing Montgomery shift"
                )
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise Exception(
                    "Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift"
                )

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder

        res = self.duplicate()

        res -= self

        mod_adder(other, res, self.inpl_adder, self.modulus)

        return res

    @gate_wrap(permeability=[1], is_qfree=True)
    def __isub__(self, other):
        if isinstance(other, int):
            other = self.encoder(other % self.modulus)
        elif isinstance(other, QuantumModulus):
            if self.m != other.m:
                raise Exception(
                    "Tried to add subtract QuantumModulus with differing Montgomery shift"
                )
        elif isinstance(other, QuantumFloat):
            if self.m != 0:
                raise Exception(
                    "Tried to subtract a QuantumFloat and QuantumModulus with non-zero Montgomery shift"
                )

        from qrisp.alg_primitives.arithmetic.modular_arithmetic import mod_adder
        from qrisp.environments import invert

        with invert():
            mod_adder(other, self, self.inpl_adder, self.modulus)

        return self

    @comparison_wrapper
    def __lt__(self, other):
        from qrisp.alg_primitives import uint_lt

        return uint_lt(self, other, self.inpl_adder)

    @comparison_wrapper
    def __gt__(self, other):
        from qrisp.alg_primitives import uint_gt

        return uint_gt(self, other, self.inpl_adder)

    @comparison_wrapper
    def __le__(self, other):
        from qrisp.alg_primitives import uint_le

        return uint_le(self, other, self.inpl_adder)

    @comparison_wrapper
    def __ge__(self, other):
        from qrisp.alg_primitives import uint_ge

        return uint_ge(self, other, self.inpl_adder)

    @comparison_wrapper
    def __eq__(self, other):
        return QuantumVariable.__eq__(self, other)

    @comparison_wrapper
    def __ne__(self, other):
        return QuantumVariable.__ne__(self, other)

    def __hash__(self):
        return QuantumFloat.__hash__(self)
