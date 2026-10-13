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

from qrisp import check_for_tracing_mode
from qrisp.alg_primitives import QFT
from qrisp.alg_primitives.arithmetic import U_g, multi_controlled_U_g
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


# Performs the modular inplace addition b += a
# where a and b don't need to have the same montgomery shift.
# The raw value of a is converted into b's representation bitwise:
# raw_b += raw_a * 2^(b.m - a.m) mod N. This bypasses b's encoder,
# so b.m is accounted for in the factor (and must not be applied twice).
# QuantumFloat operands (value = raw * 2^exponent) are treated as having
# shift -exponent.
def montgomery_addition(a, b):
    from qrisp.alg_primitives.arithmetic.jasp_arithmetic.jasp_mod_tools import (
        pow2mod,
        pow2mod_signed,
    )

    N = b.modulus
    a_m = a.m if hasattr(a, "m") else -a.exponent
    factor = pow2mod_signed(b.m - a_m, N)

    for i in jrange(a.size):
        mod_adder((factor * pow2mod(i, N)) % N, b, b.inpl_adder, N, ctrl=a[i])


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
