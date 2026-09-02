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

"""Implements Shor's integer factorization algorithm via quantum order finding."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from sympy import Rational, continued_fraction_convergents, continued_fraction_iterator

from qrisp import QuantumFloat, QuantumModulus, control, h
from qrisp.alg_primitives import QFT

if TYPE_CHECKING:
    from qrisp.interface.measurement_result import DecodedMeasurementResult
from qrisp.alg_primitives.arithmetic.modular_arithmetic import find_optimal_m, modinv

depths = []
cnot_count = []
qubits = []


def _find_optimal_a(N: int) -> list[int]:
    n = int(np.ceil(np.log2(N)))
    proposals = []

    # Search through the first O(1) possibilities to find a good a
    for a in range(2, min(100, N - 1)):
        # We only append non-trivial proposals
        if np.gcd(a, N) == 1:
            proposals.append(a)

    cost_dic = {}
    for a in proposals:
        m_values = []
        for k in range(2 * n + 1):
            inpl_multiplier = pow(a, 2**k, N)

            if inpl_multiplier == 1:
                continue

            # find_optimal_m is a function that determines the lowest possible
            # Montgomery shift for a given number. The higher the montgomery shift,
            # the more qubits and the more effort is needed.
            m_values.append(find_optimal_m(inpl_multiplier, N))
            m_values.append(find_optimal_m(modinv((-inpl_multiplier) % N, N), N))

        cost_dic[a] = sum(m_values) + max(m_values) * 1e-5

    proposals.sort(key=lambda a: cost_dic[a])

    return proposals


def _find_order(a: int, N: int, inpl_adder: Callable | None = None, mes_kwargs: dict | None = None) -> int:
    if mes_kwargs is None:
        mes_kwargs = {}
    orig_a = a
    qg = QuantumModulus(N, inpl_adder)
    qg[:] = 1
    qpe_res = QuantumFloat(2 * qg.size + 1, exponent=-(2 * qg.size + 1))
    h(qpe_res)
    for qb in qpe_res:
        with control(qb):
            qg *= a
            a = (a * a) % N
    QFT(qpe_res, inv=True, inpl_adder=inpl_adder)

    mes_res = qpe_res.get_measurement(**mes_kwargs)

    return _extract_order(mes_res, orig_a, N)


def _extract_order(mes_res: "DecodedMeasurementResult", a: int, N: int) -> int:
    # Bounds the accumulated-candidate search below, keeping it polynomial
    # (not exponential) in the number of outcomes examined even in an
    # adversarial case. Combining every candidate from every outcome
    # simultaneously, unbounded, previously made this effectively never
    # return; two or three independent QPE measurements are, in practice,
    # essentially always enough to recover the true order.
    max_candidates = 64

    accumulated_r_values: list[int] = []

    approximations = list(mes_res.keys())

    try:
        approximations.remove(0)
    except ValueError:
        pass

    while True:
        r_values = _get_r_values(approximations.pop(0))

        for r in r_values:
            if pow(a, r, N) == 1:
                return r

        # Combine this outcome's candidates against every LCM combination
        # accumulated from *all* previous outcomes so far (not just each
        # individual past outcome in isolation), so an order that only
        # emerges from combining three or more outcomes (e.g. an order of
        # 30 recovered as lcm(2, 3, 5), where no pairwise combination
        # alone divides 30) is still found.
        new_candidates = []
        for prev_r in accumulated_r_values:
            for r in r_values:
                combined = int(np.lcm(r, prev_r))
                if pow(a, combined, N) == 1:
                    return combined
                if combined not in accumulated_r_values and combined not in new_candidates:
                    new_candidates.append(combined)

        accumulated_r_values.extend(r_values)
        accumulated_r_values.extend(new_candidates)
        if len(accumulated_r_values) > max_candidates:
            accumulated_r_values = accumulated_r_values[-max_candidates:]


def _get_r_values(approx: int | float) -> list[int]:
    rationals = continued_fraction_convergents(continued_fraction_iterator(Rational(approx)))
    return [rat.q for rat in rationals if 1 < rat.q]


def shors_alg(N: int, inpl_adder: Callable | None = None, mes_kwargs: dict | None = None) -> int:
    """Performs `Shor's factorization algorithm <https://arxiv.org/abs/quant-ph/9508027>`_ on a given integer N.

    The adder used for factorization can be customized. To learn more about
    this feature, please read :ref:`QuantumModulus`

    Parameters
    ----------
    N : integer
        The integer to be factored.
    inpl_adder : callable, optional
        A function that performs in-place addition. The default is None.
    mes_kwargs : dict, optional
        A dictionary of keyword arguments for :meth:`get_measurement <qrisp.QuantumVariable.get_measurement>`.
        This especially allows you to specify an execution backend. The default is {}.

    Returns
    -------
    res : integer
        A factor of N.

    Examples
    --------
    We factor 65:

    >>> from qrisp.shor import shors_alg
    >>> shors_alg(65)
    5

    """
    if not N % 2:
        return 2

    if mes_kwargs is None:
        mes_kwargs = {}

    a_proposals = _find_optimal_a(N)

    for a in a_proposals:
        K = np.gcd(a, N)

        if K != 1:
            res = K
            break

        r = _find_order(a, N, inpl_adder, mes_kwargs)

        if r % 2:
            continue

        # gcd(x, N) == gcd(x mod N, N) for any integer x, so reducing a**(r//2)
        # modulo N before the +1 is mathematically exact and avoids computing
        # a**(r//2) in full precision, which can be an enormous number for
        # large r.
        g = int(np.gcd(pow(a, r // 2, N) + 1, N))

        if g not in [N, 1]:
            res = g
            break
    else:
        raise RuntimeError(
            f"Shor's algorithm failed to find a nontrivial factor of {N} using the given candidate bases"
        )
    return res
