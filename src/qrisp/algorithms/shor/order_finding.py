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

"""Implements the quantum part of order finding with a semiclassical phase estimation, in Jasp mode."""

# The semiclassical phase estimation itself is in qrisp.alg_primitives.iterative_qpe.

from __future__ import annotations

import math
import operator
from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array

from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import DTYPE, BigInteger, _static_digits
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import _LIMB_BITS
from qrisp.alg_primitives.iterative_qpe import _semiclassical_phase_estimation
from qrisp.core import reset
from qrisp.jasp import check_for_tracing_mode
from qrisp.qtypes import QuantumModulus

__all__ = ["semiclassical_order_finding"]

# The Montgomery multiplication needs an odd modulus, and order finding a modulus of at least 3
_MIN_MODULUS = 3


def semiclassical_order_finding(
    a: int,
    N: int,
    precision: int | None = None,
    inpl_adder: Callable | None = None,
) -> BigInteger:
    r"""Run the quantum part of order finding with a semiclassical phase estimation, and return its outcome.

    This is the quantum subroutine of Shor's algorithm: the phase estimation of the
    multiplication by $a$ modulo $N$ on a :ref:`QuantumModulus <QuantumModulus>`
    initialized to 1. Its outcome $y$ is close to $s \cdot 2^t / r$ for a random
    $s$, where $r$ is the order of $a$ modulo $N$ and $t$ is the precision, so
    that $r$ can be recovered classically, for example with continued fractions.

    Instead of a $t$-qubit phase register followed by the inverse quantum Fourier
    transform, the phase estimation uses the semiclassical Fourier transform of
    `Griffiths and Niu (1996) <https://arxiv.org/abs/quant-ph/9511007>`_, with one
    control qubit at a time following `Mosca and Ekert (1999)
    <https://arxiv.org/abs/quant-ph/9903071>`_. Step $j$ applies the multiplication
    by $a^{2^{t-1-j}} \bmod N$ controlled on a fresh qubit, rotates the qubit by the
    phase of the bits measured before, measures bit $j$ of $y$, and resets the
    qubit before releasing it. The outcome has the same distribution as with the
    full phase register. The circuit needs $t - 1$ fewer qubits, and $t$
    single-qubit rotations replace the $t (t - 1) / 2$ controlled phase gates and
    $\lfloor t / 2 \rfloor$ swaps of the inverse quantum Fourier transform. It
    requires mid-circuit measurements and rotations that depend on their outcomes.

    The classical multipliers $a^{2^k} \bmod N$ and their inverses modulo $N$,
    which the in-place multiplications need, are computed exactly with Python
    integers when the function is called, so ``a`` and ``N`` must be integers and
    cannot be traced.

    .. note::

        This function can only be called in :ref:`Jasp <jasp>` mode, for example
        within :ref:`jaspify <jaspify>`, or in the resource estimators
        :ref:`count_ops <count_ops>` and :ref:`num_qubits <num_qubits>`.

    Parameters
    ----------
    a : int
        The number whose order is found, coprime to ``N``.
    N : int
        The modulus, an odd integer larger than 2.
    precision : int, optional
        The number $t$ of bits of the outcome. The default is $2n$, where
        $n = \lceil \log_2 N \rceil$ is the number of qubits of the register.
    inpl_adder : Callable, optional
        The in-place adder of the modular multiplication, see
        :ref:`QuantumModulus <QuantumModulus>`. The default is the
        :func:`gidney_adder <qrisp.gidney_adder>`.

    Returns
    -------
    BigInteger
        The outcome $y$, with $\lceil t / 32 \rceil$ limbs. The register is reset and
        deleted before returning, so that repeated calls do not keep it allocated and
        later allocations can reuse its qubits.

    Raises
    ------
    TypeError
        If ``a``, ``N`` or ``precision`` is not an integer.
    ValueError
        If ``N`` is even or smaller than 3, if ``a`` is not coprime to ``N``, or if
        ``precision`` is smaller than 1.
    RuntimeError
        If the function is called outside of Jasp mode.

    Examples
    --------
    The order of 7 modulo 15 is $r = 4$, and $2^t$ with the default precision
    $t = 8$ is a multiple of $r$. The outcome is therefore one of the multiples
    $s \cdot 2^8 / 4 = 64 s$, each with probability $1/4$. Calling the returned
    :ref:`BigInteger <scalable_integer_type>` gives its value:

    >>> from qrisp.algorithms.shor import semiclassical_order_finding
    >>> from qrisp.jasp import jaspify
    >>> def order_finding():
    ...     return semiclassical_order_finding(7, 15)
    >>> outcome = jaspify(order_finding)()  # doctest: +SKIP
    >>> print(outcome() % 64)  # doctest: +SKIP
    0

    The resource estimators count the same circuit without simulating it. With
    the default adder, the circuit needs $3n + 2 \lceil \log_2 n \rceil + 2$
    qubits, 18 for the $n = 4$ qubits of the register. It applies one rotation per
    bit of the outcome, 8 for $t = 8$:

    >>> from qrisp.jasp import count_ops, num_qubits
    >>> print(num_qubits(meas_behavior="1")(order_finding)()["peak_allocations"])
    18
    >>> print(count_ops(meas_behavior="1")(order_finding)()["rz"])
    8

    """
    a = operator.index(a)
    N = operator.index(N)
    if N < _MIN_MODULUS or N % 2 == 0:
        raise ValueError(f"The modulus must be an odd integer larger than 2, got {N}")
    if math.gcd(a, N) != 1:
        raise ValueError(f"{a} is not coprime to {N}, so it has no order modulo {N}")
    if not check_for_tracing_mode():
        raise RuntimeError("semiclassical_order_finding can only be called in Jasp mode")

    num_register_qubits = (N - 1).bit_length()  # the size of QuantumModulus(N)
    if precision is None:
        precision = 2 * num_register_qubits
    precision = operator.index(precision)
    if precision < 1:
        raise ValueError(f"The precision must be at least 1, got {precision}")

    # The modulus and the multipliers need the limbs of the register only: the
    # multiplication widens them itself to hold its intermediate products
    num_limbs = -(-num_register_qubits // _LIMB_BITS)
    # Step j multiplies by a**(2**(t - 1 - j)) mod N. The multipliers and their
    # inverses modulo N are computed here exactly with Python integers, in NumPy
    # arrays, since the limbs of a BigInteger would be traced.
    multipliers = np.empty((precision, num_limbs), dtype=DTYPE)
    inverses = np.empty((precision, num_limbs), dtype=DTYPE)
    multiplier, inverse = a % N, pow(a, -1, N)
    for k in range(precision):
        multipliers[k] = _static_digits(multiplier, num_limbs)
        inverses[k] = _static_digits(inverse, num_limbs)
        multiplier = multiplier * multiplier % N
        inverse = inverse * inverse % N
    multipliers, inverses = jnp.asarray(multipliers), jnp.asarray(inverses)

    register = QuantumModulus(BigInteger.create_static(N, num_limbs), inpl_adder=inpl_adder)
    register[:] = 1

    def multiply_by_power(register: QuantumModulus, k: Array) -> None:
        """Multiply the register in place by ``a**(2**k) mod N``.

        Parameters
        ----------
        register : QuantumModulus
            The register.
        k : Array
            The exponent of the power of two.

        """
        register.inpl_mult(BigInteger(multipliers[k]), inverse=BigInteger(inverses[k]))

    _, digits = _semiclassical_phase_estimation(register, multiply_by_power, precision, outcome_limbs=True)
    # The phase estimation leaves the register in a state other than |0>, and a released
    # qubit must be in |0> for the interpreters that reuse it
    reset(register)
    register.delete()
    return BigInteger(digits)
