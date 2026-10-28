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

"""Implements Beauregard's QFT-based modular adder for QuantumFloats and QuantumModulus."""

import numpy as np
from jax.core import Tracer

from qrisp import check_for_tracing_mode
from qrisp.alg_primitives import QFT
from qrisp.alg_primitives.arithmetic import U_g, multi_controlled_U_g
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_bigintiger import BigInteger
from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import pow2mod_signed
from qrisp.core.gate_application_functions import cx
from qrisp.environments import conjugate, control, custom_control, invert
from qrisp.jasp import jrange
from qrisp.qtypes.quantum_bool import QuantumBool
from qrisp.qtypes.quantum_float import QuantumFloat


def qft_basis_adder(addend, target):

    if isinstance(addend, int):
        U_g(addend, target)
    elif isinstance(addend, QuantumFloat):
        if addend.signed:
            raise Exception("Signed addition not supported")
        for i in range(*addend.mshape):
            multi_controlled_U_g(target, [addend.significant(i)], 2**i)


def _is_known_even(N):
    """Return True if N is known to be even without evaluating traced values.

    Covers Python and NumPy integers as well as BigIntegers with concrete digits.
    Traced moduli cannot be inspected at trace time and return False.
    """
    if isinstance(N, (int, np.integer)):
        return int(N) % 2 == 0
    if isinstance(N, BigInteger) and not isinstance(N.digits, Tracer):
        return int(np.asarray(N.digits)[0]) % 2 == 0
    return False


# Performs the modular inplace addition b += a
# where a and b don't need to have the same montgomery shift.
# The raw value of a is converted into b's representation bitwise:
# raw_b += raw_a * 2^(b.m - a.m) mod N. This bypasses b's encoder,
# so b.m is accounted for in the per-bit constants (and must not be applied twice).
# QuantumFloat operands (value = raw * 2^exponent) are treated as having
# shift -exponent.
# Each bit's constant 2^(b.m - a.m + i) mod N is computed directly, since
# multiplying two residues first would overflow int64/BigInteger for large N.
def montgomery_addition(a, b):

    N = b.modulus
    # Negative exponents require 2 to be invertible mod N
    # (traced moduli cannot be checked and must be odd)
    if _is_known_even(N):
        raise ValueError("Addition with differing Montgomery shifts requires an odd modulus")

    a_m = a.m if hasattr(a, "m") else -a.exponent
    shift = b.m - a_m

    for i in jrange(a.size):
        # Nest the control instead of passing ctrl=a[i]: in static mode, custom_control
        # replaces an explicit ctrl with the qubit of an enclosing control environment
        with control(a[i]):
            mod_adder(pow2mod_signed(shift + i, N), b, b.inpl_adder, N)


def beauregard_adder(a, b, modulus):

    if modulus > 2**a.size:
        raise Exception("Tried to perform modular addition on QuantumFloat with too few qubits")
    if modulus == 2**a.size:
        with conjugate(QFT)(a, exec_swap=False):
            qft_basis_adder(b, a)
        return

    reduction_not_necessary = QuantumBool()
    sign = QuantumBool()

    if isinstance(b, int):
        b = b % modulus

    a = list(a) + [sign[0]]

    with conjugate(QFT)(a, exec_swap=False):
        qft_basis_adder(b, a)

        with invert():
            qft_basis_adder(modulus, a)

        with conjugate(QFT)(a, exec_swap=False, inv=True):
            cx(sign, reduction_not_necessary)

        with control(reduction_not_necessary):
            qft_basis_adder(modulus, a)

        with invert():
            qft_basis_adder(b, a)

        with conjugate(QFT)(a, exec_swap=False, inv=True):
            cx(sign, reduction_not_necessary)
            reduction_not_necessary.flip()

        qft_basis_adder(b, a)

    sign.delete()
    reduction_not_necessary.delete()


@custom_control
def mod_adder(a, b, inpl_adder, modulus, ctrl=None):

    reduction_not_necessary = QuantumBool()
    sign = QuantumBool()

    if isinstance(a, int):
        a = a % modulus

    if check_for_tracing_mode():
        b = b[:] + sign[:]
    else:
        b = list(b) + [sign[0]]

    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)

    with invert():
        inpl_adder(modulus, b)

    cx(sign, reduction_not_necessary)

    with control(reduction_not_necessary):
        inpl_adder(modulus, b)

    with invert():
        if ctrl is None:
            inpl_adder(a, b)
        else:
            with control(ctrl):
                inpl_adder(a, b)

    cx(sign, reduction_not_necessary)
    reduction_not_necessary.flip()

    if ctrl is None:
        inpl_adder(a, b)
    else:
        with control(ctrl):
            inpl_adder(a, b)

    sign.delete()
    reduction_not_necessary.delete()
