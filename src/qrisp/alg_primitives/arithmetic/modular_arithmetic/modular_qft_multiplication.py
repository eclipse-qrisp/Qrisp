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

from qrisp.qtypes.quantum_float import QuantumFloat
from qrisp.alg_primitives import QFT
from qrisp.alg_primitives.arithmetic import multi_controlled_U_g, hybrid_mult
from qrisp.core.gate_application_functions import h, cx, swap, mcx
from qrisp.environments import conjugate, control
from qrisp.alg_primitives.arithmetic.modular_arithmetic.mod_tools import (
    modinv,
    montgomery_encoder,
)


def QREDC(t, N, m):

    u = QuantumFloat(0, qs=t.qs, name="u*")

    for i in range(m):

        h(t[0])

        transfer_lsb(t, u)

        t.exp_shift(-1)
        multi_controlled_U_g(t, [u[-1]], -N / 2)

    QFT(t, inv=True, exec_swap=False)

    with conjugate(QFT)(t[:-1], exec_swap=False):

        multi_controlled_U_g(t[:-1], [t.sign()], N)

    cx(t[0], t.sign())

    sgn = t.reg.pop(-1)
    t.signed = False

    u.reg.insert(len(u), sgn)

    t.m = t.m - m

    return t, u


def find_optimal_m(b, N):
    m = 0
    n = int(np.ceil(np.log2(N)))

    while True:
        b_trial = montgomery_encoder(b, 2**m, N) % N
        if (N - 1) * b_trial < N * 2 ** (m):
            return m
        m += 1


def montgomery_mod_semi_mul(a, b, output_qg=None, permeable_if_zero=False):

    N = a.modulus

    b = b % N

    m = find_optimal_m(b, N)

    if b == 0:
        if output_qg is None:
            return a.duplicate()
        else:
            return output_qg
    if b == 1:
        if output_qg is None:
            return a.duplicate(init=True)
        else:
            output_qg += a
            return output_qg

    m = find_optimal_m(b, N)
    b = montgomery_encoder(b, 2**m, N) % N

    if output_qg is None:
        t = QuantumFloat(a.size + m, signed=True)

        h(t)
    else:
        if output_qg.modulus != N:
            raise Exception("Output QuantumModulus has incompatible modulus")

        output_qg.extend(m, 0)
        output_qg.add_sign()

        QFT(output_qg, exec_swap=False)

        t = output_qg

    for i in range(a.size):
        multi_controlled_U_g(t, [a[i]], (2**i * b))

    from qrisp import QuantumModulus

    t.__class__ = QuantumModulus
    t.modulus = a.modulus
    t.m = a.m + m
    t.inpl_adder = a.inpl_adder

    return montgomery_red(t, a, b, N, m, permeable_if_zero=permeable_if_zero)


def montgomery_red(t, a, b, N, m, permeable_if_zero=False):

    t, u = QREDC(t, N, m)

    with conjugate(QFT)(u, exec_swap=False):
        if isinstance(b, QuantumFloat):

            hybrid_mult(
                a,
                b,
                output_qf=u,
                init_op=None,
                terminal_op=None,
                cl_factor=-modinv(N, 2 ** (m + 1)),
            )

        else:
            for i in range(len(a)):
                multi_controlled_U_g(u, [a[i]], -((2**i * b)) * modinv(N, 2 ** (m + 1)))

    if permeable_if_zero:
        mcx(list(a) + [t[0]], u[-1], ctrl_state="0" * len(a) + "1", method="balauca")

    u.delete(verify=False)

    return t


from qrisp import merge


def montgomery_mod_mul(a, b, output_qg=None):

    m = int(np.ceil(np.log2((a.modulus - 1) ** 2) + 1)) - a.size

    if a.modulus != b.modulus:
        raise Exception("Tried to multiply two QuantumModulus with differing modulus")

    if output_qg is None:
        t = QuantumFloat(a.size + m, signed=True)
        h(t)

    else:
        if output_qg.modulus != a.modulus:
            raise Exception("Output QuantumModulus has incompatible modulus")

        merge(output_qg.qs, a.qs)
        output_qg.extend(m, 0)
        output_qg.add_sign()
        output_qg.reg.insert(0, output_qg.reg.pop(-1))

        QFT(output_qg, exec_swap=False)

        t = output_qg

    t = hybrid_mult(a, b, init_op=None, terminal_op=None, output_qf=t)

    from qrisp import QuantumModulus

    t.__class__ = QuantumModulus
    t.modulus = a.modulus
    t.m = a.m + b.m
    t.inpl_adder = a.inpl_adder

    t = montgomery_red(t, a, b, a.modulus, m)

    return t


from qrisp import invert
from qrisp.environments import custom_control


@custom_control
def qft_semi_cl_inpl_mult(a, X, ctrl=None, treat_invalid=False):

    X = X % a.modulus

    if X == 0:
        raise Exception(
            "Tried to perform in-place multiplication with 0 (not invertible)"
        )
    if X == 1:
        return a

    tmp = a.duplicate(qs=a.qs)

    from qrisp import multi_measurement, less_than, redirect_qfunction, QuantumModulus

    if treat_invalid:
        a.__class__ = QuantumFloat
        reduced = a < a.modulus
        a.__class__ = QuantumModulus
        ctrl = [ctrl, reduced[0]]

    if ctrl is not None:
        with control(ctrl, invert=True):
            swap(tmp, a)

    tmp = montgomery_mod_semi_mul(
        a, X, output_qg=tmp, permeable_if_zero=ctrl is not None
    )

    if ctrl is not None:
        with control(ctrl, invert=True):
            swap(tmp, a)

    with invert():
        a = montgomery_mod_semi_mul(
            tmp,
            modinv(X, a.modulus) % a.modulus,
            output_qg=a,
            permeable_if_zero=ctrl is not None,
        )

    if ctrl is not None:
        with control(ctrl, invert=False):
            swap(tmp, a)
    else:
        swap(tmp, a)

    tmp.delete(verify=False)

    if treat_invalid:
        a.__class__ = QuantumFloat
        redirect_qfunction(less_than)(a, a.modulus, target=reduced)
        a.__class__ = QuantumModulus
        reduced.delete(verify=False)

    return a


def transfer_lsb(from_qv, to_qv):
    lsb = from_qv.reg.pop(0)
    from_qv.exponent += 1

    to_qv.reg.insert(len(to_qv), lsb)
