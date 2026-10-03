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

# The phase estimation of order finding can reuse a single control qubit
# (Griffiths and Niu, Phys. Rev. Lett. 76, 3228, 1996). Step j applies the
# multiplication by a^(2^(t-1-j)) mod N controlled on a fresh qubit, removes the
# phase of the bits measured before with a rotation, and measures bit j of the
# outcome. The outcome has the same distribution as the measurement of a t-qubit
# phase register after the inverse QFT, while the circuit needs t - 1 fewer qubits.

from __future__ import annotations

import math
import operator
from collections.abc import Callable
from typing import cast

import jax.numpy as jnp
import numpy as np
from jax import Array

from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.alg_primitives.iterative_qpe import _semiclassical_phase_estimation
from qrisp.jasp import check_for_tracing_mode
from qrisp.qtypes import QuantumModulus

__all__ = ["semiclassical_order_finding"]

# Number of bits of a BigInteger limb
_LIMB_BITS = 32
# The Montgomery multiplication needs an odd modulus, and order finding a modulus of at least 3
_MIN_MODULUS = 3


def _to_limbs(value: int, num_limbs: int) -> np.ndarray:
    """Return the little-endian 32-bit limbs of a non-negative Python integer, as in a BigInteger.

    Parameters
    ----------
    value : int
        The integer, smaller than ``2**(32 * num_limbs)``.
    num_limbs : int
        The number of limbs.

    Returns
    -------
    np.ndarray
        The limbs, of dtype ``uint32``.

    """
    mask = (1 << _LIMB_BITS) - 1
    return np.array([(value >> (_LIMB_BITS * i)) & mask for i in range(num_limbs)], dtype=np.uint32)


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
    transform, the phase estimation uses one control qubit at a time (`Griffiths
    and Niu, 1996 <https://arxiv.org/abs/quant-ph/9511007>`_): step $j$ applies the
    multiplication by $a^{2^{t-1-j}} \bmod N$ controlled on a fresh qubit, rotates
    the qubit by the phase of the bits measured before, and measures bit $j$ of
    $y$. The outcome has the same distribution as with the full phase register,
    while the circuit needs $t - 1$ fewer qubits and $t$ single-qubit rotations
    instead of the $t (t - 1) / 2$ controlled phase gates of the inverse quantum
    Fourier transform. It requires mid-circuit measurements and rotations that
    depend on their outcomes.

    The classical multipliers $a^{2^k} \bmod N$ are computed exactly before the
    circuit is traced, so ``a`` and ``N`` must be Python integers.

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
        :meth:`gidney_adder <qrisp.gidney_adder>`.

    Returns
    -------
    BigInteger
        The outcome $y$, with $\lceil t / 32 \rceil$ limbs.

    Raises
    ------
    ValueError
        If ``N`` is even or smaller than 3, if ``a`` is not coprime to ``N``, or if
        ``precision`` is smaller than 1.
    RuntimeError
        If the function is called outside of Jasp mode.

    Examples
    --------
    The order of 7 modulo 15 is $r = 4$, and $2^t$ with the default precision
    $t = 8$ is a multiple of $r$. The outcome is therefore exactly one of the
    multiples $s \cdot 2^8 / 4$, each with probability $1/4$:

    ::

        from qrisp.algorithms.shor import semiclassical_order_finding
        from qrisp.jasp import jaspify

        @jaspify
        def main():
            return semiclassical_order_finding(7, 15)

        outcome = main()
        print(outcome())  # The value of the BigInteger
        # 0, 64, 128 or 192

    The resource estimators count the circuit without simulating it. With the
    default adder, the circuit needs $3n + 2 \lceil \log_2 n \rceil + 2$ qubits:

    ::

        from qrisp.jasp import num_qubits

        def main():
            return semiclassical_order_finding(7, 15)

        print(num_qubits(meas_behavior="1")(main)()["peak_allocations"])
        # 18

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

    # The classical numbers share one width, large enough for the modulus and the outcome
    num_limbs = -(-max(precision, 2 * num_register_qubits) // _LIMB_BITS)
    # Step j multiplies by a**(2**(t - 1 - j)) mod N, computed here with Python integers
    multipliers = np.empty((precision, num_limbs), dtype=np.uint32)
    multiplier = a % N
    for k in range(precision):
        multipliers[k] = _to_limbs(multiplier, num_limbs)
        multiplier = multiplier * multiplier % N
    multipliers = jnp.asarray(multipliers)

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
        register *= BigInteger(multipliers[k])

    num_outcome_limbs = -(-precision // _LIMB_BITS)
    _, outcome = _semiclassical_phase_estimation(register, multiply_by_power, precision, num_limbs=num_outcome_limbs)
    # The outcome is returned, since num_limbs is positive
    return cast("BigInteger", outcome)
