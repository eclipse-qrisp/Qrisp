# ********************************************************************************
# * Copyright (c) 2024 the Qrisp authors
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

"""Implements a faster trotterization path for "Ising-type" Hamiltonians
(only identity,single-qubit Pauli, and two-qubit Z*Z terms)"""

import jax.numpy as jnp

from qrisp import IterationEnvironment, invert, merge, rx, ry, rz, rzz
from qrisp.jasp import check_for_tracing_mode, jrange


def is_flat_ising_operator(H):
    r"""
    Check whether a QubitOperator contains only identity, single-qubit
    X/Y/Z, or two-qubit Z*Z terms -- i.e. whether it can be simulated with
    :func:`fast_trotterization`'s native-gate fast path instead of the
    general ``QubitOperator.trotterization()``.

    Parameters
    ----------
    H : :ref:`QubitOperator`
        The operator to check.

    Returns
    -------
    bool
        True if every term of H is the identity, a single-qubit Pauli, or a
        two-qubit Z*Z term.

    """
    for term in H.terms_dict:
        factor_dict = term.factor_dict
        n = len(factor_dict)
        if n == 0:
            continue
        elif n == 1:
            if next(iter(factor_dict.values())) not in ("X", "Y", "Z"):
                return False
        elif n == 2:
            if set(factor_dict.values()) != {"Z"}:
                return False
        else:
            return False
    return True


def _emit_flat_trotter_step(H, qarg, t, steps, forward_evolution):
    sign = 1 if forward_evolution else -1
    for term, term_coeff in H.terms_dict.items():
        factor_dict = term.factor_dict
        if len(factor_dict) == 0:
            continue
        angle = 2 * sign * jnp.real(term_coeff) * t / steps
        if len(factor_dict) == 1:
            ((index, pauli),) = factor_dict.items()
            if pauli == "X":
                rx(angle, qarg[index])
            elif pauli == "Y":
                ry(angle, qarg[index])
            else:
                rz(angle, qarg[index])
        else:
            i, j = sorted(factor_dict.keys())
            rzz(angle, qarg[i], qarg[j])


def _flat_trotterization(H, order=1, forward_evolution=True):
    def trotter_step(qarg, t, steps):
        _emit_flat_trotter_step(H, qarg, t, steps, forward_evolution)

    def U(qarg, t=1, steps=1, iter=1):
        if check_for_tracing_mode():
            for i in jrange(iter * steps):
                if order == 1:
                    trotter_step(qarg, t, steps)
                elif order == 2:
                    trotter_step(qarg, t, steps * 2)
                    with invert():
                        trotter_step(qarg, -t, steps * 2)
        else:
            merge([qarg])
            with IterationEnvironment(qarg.qs, iter * steps):
                if order == 1:
                    trotter_step(qarg, t, steps)
                elif order == 2:
                    trotter_step(qarg, t, steps * 2)
                    with invert():
                        trotter_step(qarg, -t, steps * 2)

    return U


def fast_trotterization(H, order=1, method="commuting_qw", forward_evolution=True):
    r"""
    Drop-in replacement for :meth:`QubitOperator.trotterization
    <qrisp.operators.qubit.QubitOperator.trotterization>` that emits native
    ``rx``/``ry``/``rz``/``rzz`` gates directly -- with no per-term
    ``QuantumEnvironment`` and therefore no extra QuantumSession merging --
    whenever ``H`` is an "Ising-type" operator (see
    :func:`is_flat_ising_operator`). For any other operator (containing
    ladder operators, projectors, 3+-qubit terms, or non-Z 2-qubit terms),
    this transparently falls back to ``H.trotterization(...)``, so it is
    always at least as general and always at least as correct as calling
    ``.trotterization()`` directly.

    Parameters
    ----------
    H : :ref:`QubitOperator`
        The Hamiltonian to Trotterize.
    order : int, optional
        The order of the Suzuki-Trotter formula. The default is 1.
    method : str, optional
        Forwarded to ``H.trotterization()`` if the fast path isn't taken.
        The default is ``"commuting_qw"``.
    forward_evolution : bool, optional
        If False, simulates $U(t)^\\dagger$ instead of $U(t)$. The default is True.

    Returns
    -------
    callable
        A function ``U(qarg, t=1, steps=1, iter=1)`` with the exact same
        signature and semantics as the function returned by
        ``QubitOperator.trotterization()``.

    """
    if is_flat_ising_operator(H):
        return _flat_trotterization(H, order=order, forward_evolution=forward_evolution)
    return H.trotterization(order=order, method=method, forward_evolution=forward_evolution)
